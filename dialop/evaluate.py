from copy import deepcopy
import json
import os
import pathlib
import time
from typing import Optional, Literal
from itertools import cycle

import numpy as np
import argparse
from ruamel.yaml import YAML
from rich import print
from rich.console import Console
import sys
console = Console()

from envs import (
    PlanningEnv,
    OptimizationEnv,
    MediationEnv,
    WordLimit,
    ForceProposal,
    AsymmetricForceProposal
)
from players import (
    LLMPlayer,
    HumanPlayer,
    DryRunPlayer,
    OutOfContextError
)
from utils import Logger, retry, count_words
from metrics import make_exp_name, aggregate_metrics, write_to_wandb

FPATH = pathlib.Path(os.path.dirname(os.path.realpath(__file__)))
RESDIR = pathlib.Path("/home/andyjliu/andy-fa25/dialop/results/")
DATADIR = pathlib.Path("/home/andyjliu/andy-fa25/dialop/data/")

GAME_CLSS = {
    "matching": OptimizationEnv,
    "itinerary": PlanningEnv,
    "mediation": MediationEnv,
}

class ResampleError(Exception):
    pass

def selfplay(
    game_cls,
    games,
    samples_per_game,
    resume,
    end
):
    for game_idx, game in enumerate(games[resume:end]):
#        data = game["games"][0]
        original_log = game["action_log"]
        data = deepcopy(game)
        # Clear action log so env doesn't initialize with a message history
        data["action_log"] = []
        if game_cls == OptimizationEnv:
            score = data["proposal_reward"]
            score_norm = data["result"]["norm"]
        else:
#            score = data["action_log"][-3]["scores"]["total"]
            score = data["result"]["score"]
            score_norm  = data["result"]["norm"]
        metadata = {
            "hh_turns": len(original_log),
            "hh_words": count_words(original_log),
            "hh_score": score,
            "hh_score_norm": score_norm,
        }
        for sidx in range(samples_per_game):
            name = f"{game_idx + resume}_{sidx}"
            yield data, name, metadata

def prompted_selfplay(
    game_cls,
    games,
    samples_per_game,
    resume,
    end,
):
    for game_idx, game in enumerate(games[resume:end]):
        if game_cls == OptimizationEnv:
            data = deepcopy(game)
            original_log = game["action_log"]
        elif game_cls == PlanningEnv:
            data = deepcopy(game)
            original_log = game["action_log"]
        elif game_cls == MediationEnv:
            data = deepcopy(game)
            original_log = game["action_log"]

        if game_cls == PlanningEnv:
            try:
                score = data["action_log"][-3]["scores"]["total"]
            except:
                for turn in range(0, len(data["action_log"])):
                    if data["action_log"][turn]["type"] == "proposal":
                        score = data["action_log"][turn]["scores"]["total"]
        elif game_cls == OptimizationEnv:
            score = data["proposal_reward"]
        elif game_cls == MediationEnv:
            score = data["result"]["score"]

        total_word_count = count_words(original_log)
        prefix_word_counts = []
        for turn in range(0, len(data["action_log"])):
            num_words = count_words(original_log[:turn])
            prefix_word_counts.append(num_words)
        # Get turns closest to 25%, 50%, 75% of the way through the game:
        turn_idxs = []
        # for pct in [0.25, 0.5, 0.75]:
        for pct in [0.5, 0.75]:
            turn_idxs.append(
                np.argmin(np.abs(np.array(prefix_word_counts) - pct * total_word_count))
            )
        # Get index of final proposal:
        proposal_idx = None
        for turn in range(0, len(data["action_log"])):
            if data["action_log"][turn]["type"] == "proposal":
                proposal_idx = turn
        if proposal_idx is None:
            raise ValueError("Game doesn't include a proposal")
        turn_idxs.append(proposal_idx)

        # import pdb; pdb.set_trace()
        # for turn in range(0, len(data["action_log"])):
        names = ["50", "75", "end"]
        # for turn in range(2, len(data["action_log"]) // 2):
        #     end = 2 * (turn + 1)
        for name, end in zip(names, turn_idxs):
            # if end >= len(game["games"][0]["action_log"]): continue
            data["action_log"] = original_log[:end]
            metadata = {
                "initialized_turns": len(data["action_log"]),
                "initialized_words": count_words(data["action_log"]),
                "hh_turns": len(original_log),
                "hh_words": count_words(original_log),
                "hh_score": score,
                "hh_score_norm": data["result"]["norm"],
            }
            for sidx in range(samples_per_game):
                name = f"{game_idx + resume}_start{len(data['action_log'])}_{name}_{sidx}"
                yield data, name, metadata

# @retry(allowed_exceptions=[OutOfContextError, ResampleError])
def run(
    game_cls,
    data,
    metadata,
    player_ctor,
    env_ctor,
    logfile,
    use_word_limit=False,
    max_length=3
):
    # Create players.
    players = player_ctor()
    # Create env.
    env = env_ctor()
    # TODO: make api the same
    if use_word_limit:
        obss = env.reset(word_limit=metadata["hh_words"],
                         game_state=data)
    else:
        obss = env.reset(game_state=data)

    # Log initial info.
    log = Logger(logfile)
    for pname, player in players.items():
        log.write(
            f"{pname} params",
            json.dumps(getattr(player, 'model_kwargs', {}), indent=2))
        log.write(f"{pname} prompt", player.prompt)
    if game_cls == PlanningEnv:
        if env.query_executor == "gpt":
            log.write(f"Query Executor Prompt", env.search.prompt)
        else:
            log.write("Using deterministic query executor.")

    # Env loop.
    t = 0
    player_cycle = cycle(players.keys())
    if game_cls == MediationEnv:
        while not obss["done"] and t < max_length:
            console.rule("environment obs")
            console.print(obss)
            [player.observe(obss[pname]) for pname, player in players.items()]
            for pname in players:
                log.log(key=pname, value=obss[pname], title=f"obs t={t}")

            # Cycle through players, letting them speak if it's their turn
            next_player = next(player_cycle)
            while next_player not in obss["turn_players"]:
                next_player = next(player_cycle)
            resample = True
            resample_cnt = 0
            while resample and resample_cnt < 3:
                if resample_cnt >= 1:
                    console.print("INVALID: resampling...", style="bold red")
                stepped = False
                while not stepped:
                    resp = players[next_player].respond()
                    stepped = True
                log.log(
                    key=next_player,
                    value=resp,
                    title=f"generate t={t} try={resample_cnt}"
                )
                stepped = False
                while not stepped:
                    obss, resample = env.step(resp, next_player)
                    stepped = True
                resample_cnt += 1
            t += 1
    else:
        while not obss["done"] and t < max_length:
            console.rule("environment obs")
            console.print(obss)
            [player.observe(obss[pname]) for pname, player in players.items()]
            for pname in players:
                log.log(key=pname, value=obss[pname], title=f"obs t={t}")
            resample = True
            resample_cnt = 0
            while resample and resample_cnt < 3:
                if resample_cnt >= 1:
                    console.print("INVALID: resampling...", style="bold red")
                resp = players[obss["turn_player"]].respond()
                log.log(
                    key=obss["turn_player"],
                    value=resp,
                    title=f"generate t={t} try={resample_cnt}"
                )
                obss, resample = env.step(resp)
                resample_cnt += 1
            t += 1

    if resample_cnt >= 3:
        print("Resampled too many times.")
        raise ResampleError()

    log.flush()
    for pname, player in players.items():
        log.flush_key(pname, title=f"{pname} Log")
        log.write(f"Final {pname} Prompt", player.prompt)
    result = {**obss, **metadata, "t": t,
              "num_turns": len(env.game.action_log),
              "num_words": count_words(env.game.action_log)}
    log.write("Result", json.dumps(result))
    log.flush()
    log.close()

def main():
    parser = argparse.ArgumentParser(description="Evaluate collaborative gym experiments.")
    parser.add_argument('--exp_name', type=str, required=True, help='Experiment name')
    parser.add_argument('--game', type=str, choices=['matching', 'itinerary', 'mediation'], required=True, help='Game type')
    parser.add_argument('--mode', type=str, choices=['selfplay', 'prompted_sp'], required=True, help='Evaluation mode')
    parser.add_argument('--resume', type=int, default=0, help='Resume index')
    parser.add_argument('--end', type=int, default=1000, help='End index')
    parser.add_argument('--samples_per_game', type=int, default=1, help='Samples per game')
    parser.add_argument('--temperature', type=float, default=0.1, help='Model temperature')
    parser.add_argument('--dry_run', type=lambda x: (str(x).lower() == 'true'), default=False, help='Dry run mode')
    parser.add_argument('--use_word_limit', type=lambda x: (str(x).lower() == 'true'), default=False, help='Use word limit')
    parser.add_argument('--model', type=str, default=None, help='Model name')
    parser.add_argument('--wandb', type=lambda x: (str(x).lower() == 'true'), default=False, help='Use wandb')
    parser.add_argument('--wandb_project', type=str, default='dialop', help='Wandb project name')

    args = parser.parse_args()

    exp_name = args.exp_name
    game = args.game
    mode = args.mode
    resume = args.resume
    end = args.end
    samples_per_game = args.samples_per_game
    temperature = args.temperature
    dry_run = args.dry_run
    use_word_limit = args.use_word_limit
    model = args.model
    wandb = args.wandb
    wandb_project = args.wandb_project

    game_cls = GAME_CLSS[game]
    EXP_DIR = RESDIR / game
    if game_cls == OptimizationEnv:
        DATA_PATH = DATADIR / "optimization.jsonl"
    elif game_cls == PlanningEnv:
        DATA_PATH = DATADIR / "itinerary.jsonl"
    elif game_cls == MediationEnv:
        DATA_PATH = DATADIR / "mediation.jsonl"

    os.makedirs(EXP_DIR / exp_name, exist_ok=True)
    with open(DATA_PATH) as f:
        games = []
        for line in f:
            games.append(json.loads(line))

    # Create generator for eval mode.
    if mode == "selfplay":
        gen = selfplay(game_cls, games, samples_per_game, resume, end)
    elif mode == "prompted_sp":
        gen = prompted_selfplay(game_cls, games, samples_per_game, resume, end)
    else:
        raise NotImplementedError()

    def create_players():
        print("Initializing players...")
        # Create prompts.
        if game_cls == OptimizationEnv:
            with open(FPATH / "prompts" / "optimization.txt") as f:
                matching_prompt = f.read()
        elif game_cls == PlanningEnv:
            with open(FPATH / "prompts" / "planning_agent.txt") as f:
                agent_prompt = f.read()
            with open(FPATH / "prompts" / "planning_user.txt") as f:
                user_prompt = f.read()
        elif game_cls == MediationEnv:
            with open(FPATH / "prompts" / "mediation_agent.txt") as f:
                agent_prompt = f.read()
            with open(FPATH / "prompts" / "mediation_user0.txt") as f:
                user0_prompt = f.read()
            with open(FPATH / "prompts" / "mediation_user1.txt") as f:
                user1_prompt = f.read()

        if game_cls == OptimizationEnv:
            p1, p2 = "player-1", "player-2"
            p1_prompt, p2_prompt = matching_prompt, matching_prompt
            optional1 = p1_prompt[
                p1_prompt.index("EXAMPLE 1"):]
            optional1 = optional1[:optional1.index("EXAMPLE 2")]
            optional2 = p2_prompt[
                p2_prompt.index("EXAMPLE 2"):]
            optional2 = optional2[:optional2.index("EXAMPLE 2")]
        elif game_cls == PlanningEnv:
            p1, p2 = "agent", "user"
            p1_prompt, p2_prompt = agent_prompt, user_prompt
            optional1, optional2 = None, None
        elif game_cls == MediationEnv:
            p1, p2, p3 = "user0", "user1", "agent"
            optional = agent_prompt[agent_prompt.index("User 0 Information"):]
            optional = optional[:optional.index("\n\n") + 2]

        if dry_run:
            assert game_cls != MediationEnv
            players = {p1: DryRunPlayer(p1_prompt, p1, console),
                       p2:  DryRunPlayer(p2_prompt, p2, console)}
        elif game_cls == MediationEnv:
            players = {p1: LLMPlayer(user0_prompt, p1, console,
                                     model=model,
                                     model_kwargs={"temperature": temperature}),
                       p2:  LLMPlayer(user1_prompt, p2, console,
                                      model=model,
                                      model_kwargs={"temperature": temperature}),
                       p3:  LLMPlayer(agent_prompt, p3, console,
                                      prefix="\nYou to",
                                      optional=optional,
                                      model=model,
                                      model_kwargs={"temperature": temperature})}
        else:
            players = {p1: LLMPlayer(p1_prompt, p1, console,
                                     optional=optional1,
                                     model=model,
                                     model_kwargs={"temperature": temperature}),
                       p2:  LLMPlayer(p2_prompt, p2, console,
                                      optional=optional2,
                                      model=model,
                                      model_kwargs={"temperature": temperature})}
        return players

    def create_env():
        print("Initializing envs...")
        if game_cls == OptimizationEnv:
            env = OptimizationEnv()
            if use_word_limit:
                env = ForceProposal(env, ["player-1", "player-2"])
        elif game_cls == PlanningEnv:
            env = PlanningEnv(query_executor="gpt")
            if use_word_limit:
                env = AsymmetricForceProposal(env, ["agent"])
        elif game_cls == MediationEnv:
            env = MediationEnv()
            if use_word_limit:
                env = AsymmetricForceProposal(env, ["agent"])
        return env

    if dry_run:
        max_length = 15
    elif game_cls == MediationEnv:
        max_length = 45
    else:
        max_length = 30

    # Evaluate.
    times = []
    for i, (data, fname, metadata) in enumerate(gen):
        if (EXP_DIR / exp_name / f"{fname}.out").exists():
            continue
        if not dry_run and i % 20 == 1:
            print(f"Sleeping... {np.mean(times):.1f}")
            time.sleep(30)
            pass
        print(fname)

        start = time.time()
        run(
            game_cls,
            data,
            metadata,
            create_players,
            create_env,
            EXP_DIR / exp_name /f"{fname}.out",
            use_word_limit=use_word_limit,
            max_length=max_length,
        )
        elapsed = (time.time() - start) / 60
        times.append(elapsed)
        print(f" == Finished {i} {elapsed:.1f} == ")

    if not exp_name:
        exp_name = make_exp_name(exp_name)
    avg_metrics = aggregate_metrics(EXP_DIR, exp_name)
    if wandb:
        args_dict = vars(args)
        write_to_wandb(avg_metrics, args_dict, wandb_project=wandb_project, exp_name=exp_name)
    
    exit()

if __name__ == "__main__":
    main()
