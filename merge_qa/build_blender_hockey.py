"""Build hockey merged QA entries from per-video hockey JSONs.

Generates 3 task entries per video:
  - hockey_own_goal: numerical (red_own_goals + blue_own_goals)
  - hockey_score: MCQ with score combinations summing to games_played
  - hockey_longest_game: MCQ with Game N choices (skipped if games_played == 1)

Usage:
  python build_hockey_merged_qa.py 5sec
  python build_hockey_merged_qa.py 10sec
  python build_hockey_merged_qa.py 20sec
  python build_hockey_merged_qa.py all
"""

import json
import os
import random
import sys

DATA_ROOT = "/Users/sihyun/Desktop/Research/projects/NYU/data"
LETTERS = "ABCD"
SEED = 42

DURATIONS = ["5sec", "10sec", "20sec"]


def make_score_choices(games_played, correct_score, rng):
    """All possible scores summing to games_played, pick correct + 3 random distractors."""
    all_scores = [f"{r}-{games_played - r}" for r in range(games_played + 1)]
    wrong = [s for s in all_scores if s != correct_score]
    distractors = rng.sample(wrong, min(3, len(wrong)))
    while len(distractors) < 3:
        extra = f"{rng.randint(0, games_played + 1)}-{rng.randint(0, games_played + 1)}"
        if extra != correct_score and extra not in distractors:
            distractors.append(extra)
    choices = [correct_score] + distractors
    rng.shuffle(choices)
    return choices


def make_game_choices(games_played, longest_game_idx, rng):
    """Choices = ['Game 1', 'Game 2', ..., 'Game N'] shuffled."""
    choices = [f"Game {i + 1}" for i in range(games_played)]
    rng.shuffle(choices)
    return choices


def build_entries(qa_dir, video_prefix):
    """Read all hockey JSONs and build 3 task entries lists."""
    rng = random.Random(SEED)

    own_goal_entries = []
    score_entries = []
    longest_entries = []

    qa_files = sorted(f for f in os.listdir(qa_dir) if f.endswith(".json"))

    for qf in qa_files:
        with open(os.path.join(qa_dir, qf)) as f:
            orig = json.load(f)

        vid_id = orig["video_id"]
        a = orig["answer"]
        vp = f"{video_prefix}/video_{vid_id}.mp4"

        # 1. hockey_own_goal (numerical)
        total_own = a["red_own_goals"] + a["blue_own_goals"]
        own_goal_entries.append({
            "video_id": vid_id,
            "video_path": vp,
            "question": "How many own goals were committed in total across all games?",
            "answer": str(total_own),
            "answer_index": None,
            "choices": None,
        })

        # 2. hockey_score (MCQ)
        games = a["games_played"]
        correct_score = a["final_score"]
        choices = make_score_choices(games, correct_score, rng)
        answer_index = choices.index(correct_score)
        answer_letter = LETTERS[answer_index]

        base_q = "What is the final score (red-blue) across all games?"
        mcq_lines = "\n".join(f"({LETTERS[j]}) {c}" for j, c in enumerate(choices))
        question = f"{base_q}\n\n{mcq_lines}\n\nPlease answer with the letter (A, B, C, or D)."

        score_entries.append({
            "video_id": vid_id,
            "video_path": vp,
            "question": question,
            "answer": answer_letter,
            "answer_index": answer_index,
            "choices": choices,
        })

        # 3. hockey_longest_game (MCQ) — skip if only 1 game
        if games > 1:
            durations = a["game_durations_seconds"]
            longest_idx = durations.index(max(durations)) + 1
            game_choices = make_game_choices(games, longest_idx, rng)
            correct_game = f"Game {longest_idx}"
            game_answer_index = game_choices.index(correct_game)
            game_answer_letter = LETTERS[game_answer_index]

            base_q = "Which game had the longest duration?"
            mcq_lines = "\n".join(f"({LETTERS[j]}) {c}" for j, c in enumerate(game_choices))
            question = f"{base_q}\n\n{mcq_lines}\n\nPlease answer with the letter (A, B, C, or D)."

            longest_entries.append({
                "video_id": vid_id,
                "video_path": vp,
                "question": question,
                "answer": game_answer_letter,
                "answer_index": game_answer_index,
                "choices": game_choices,
            })

    return own_goal_entries, score_entries, longest_entries


def update_merged_qa(duration):
    """Add hockey tasks to the merged QA file for the given duration."""
    merged_path = f"{DATA_ROOT}/merged_qa/{duration}_stretch_0p2s.json"
    qa_dir = f"{DATA_ROOT}/duration_estimation/hockey/{duration}_stretch_0p2s"
    video_prefix = f"/nas2/longvideo_eval/blender/data/duration_estimation/hockey/{duration}"

    if not os.path.exists(qa_dir):
        print(f"SKIP {duration}: {qa_dir} not found")
        return

    if not os.path.exists(merged_path):
        print(f"SKIP {duration}: {merged_path} not found")
        return

    own_goal, score, longest = build_entries(qa_dir, video_prefix)

    with open(merged_path) as f:
        merged = json.load(f)

    merged["data"]["hockey_own_goal"] = own_goal
    merged["data"]["hockey_score"] = score
    if longest:
        merged["data"]["hockey_longest_game"] = longest
    elif "hockey_longest_game" in merged["data"]:
        del merged["data"]["hockey_longest_game"]

    # Write compact (no indent) to avoid pyarrow block boundary issues
    with open(merged_path, "w") as f:
        json.dump(merged, f)

    print(f"{duration}_stretch_0p2s.json:")
    print(f"  hockey_own_goal: {len(own_goal)} entries")
    print(f"  hockey_score: {len(score)} entries (sample choices: {score[0]['choices']})")
    if longest:
        print(f"  hockey_longest_game: {len(longest)} entries")
    else:
        print(f"  hockey_longest_game: SKIPPED (single-game videos)")


def main():
    arg = sys.argv[1] if len(sys.argv) > 1 else "all"
    targets = DURATIONS if arg == "all" else [arg]

    for dur in targets:
        if dur not in DURATIONS:
            print(f"Unknown duration: {dur}. Choose from {DURATIONS} or 'all'.")
            continue
        update_merged_qa(dur)
        print()


if __name__ == "__main__":
    main()