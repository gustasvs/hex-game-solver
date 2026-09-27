from collections import Counter
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import arena as arena_module
from arena import _opening_schedule
from game_logic.hex import HexBoardState
from settings import HEX_BOARD_SIZE


class ArenaOpeningScheduleTest(unittest.TestCase):
    def test_every_opening_is_played_once_with_each_color_assignment(self):
        competitor_a = ("A", None)
        competitor_b = ("B", None)
        openings = HexBoardState().get_legal_moves()

        schedule = list(_opening_schedule(competitor_a, competitor_b, openings))

        self.assertEqual(len(schedule), 2 * HEX_BOARD_SIZE**2)
        self.assertEqual(
            Counter(move.idx for _, _, move in schedule),
            Counter({index: 2 for index in range(HEX_BOARD_SIZE**2)}),
        )
        for opening_index in range(0, len(schedule), 2):
            first = schedule[opening_index]
            second = schedule[opening_index + 1]
            self.assertEqual(first[:2], (competitor_a, competitor_b))
            self.assertEqual(second[:2], (competitor_b, competitor_a))
            self.assertIs(first[2], second[2])

    def test_arena_promotes_and_matches_edges_by_integer_move_index(self):
        class QuietProgress:
            def __init__(self, *_args, **_kwargs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def set_description(self, *_args, **_kwargs):
                pass

            def update(self, *_args, **_kwargs):
                pass

        class TwoPlyMCTS:
            created_move_indices = []

            def __init__(self, root, model, max_iterations):
                self.root = root

            def run(self):
                edge = self.root.create_edge(next(self.root.iter_legal_move_indices()))
                self.created_move_indices.append(edge.move_idx)
                if self.root.move_index >= 2:
                    edge.child.stored_terminal = True
                    edge.child.stored_terminal_outcome = 1
                return self.root

        with (
            patch("arena.MCTS", TwoPlyMCTS),
            patch("arena.tqdm", QuietProgress),
            patch("builtins.print"),
        ):
            arena_module.arena(model_1=object(), mcts_iterations=1)

        self.assertEqual(len(TwoPlyMCTS.created_move_indices), 4 * HEX_BOARD_SIZE**2)
        self.assertTrue(all(isinstance(move_idx, int) for move_idx in TwoPlyMCTS.created_move_indices))


if __name__ == "__main__":
    unittest.main()
