import copy
import pathlib
import sys
import unittest


HERE = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

import update_paper_counts as updater  # noqa: E402


class PaperInventoryTests(unittest.TestCase):
    @staticmethod
    def counts():
        return {
            "sweep1_jobs": 1200,
            "sweep1_leaves": 900,
            "sweep2_jobs": 240,
            "sweep2_leaves": 1400,
            "region1_jobs": 1404,
            "region1_angular_leaves": 4542,
            "region1_radial_pieces": 16543,
            "block3a_cells": 247,
            "block3a_leaves": 263,
            "block3bc": {
                "b_pos": {"top_cells": 24, "leaves": 24},
                "b_neg": {"top_cells": 331, "leaves": 400},
                "c": {"top_cells": 16, "leaves": 20},
            },
            "k_nodes": 59,
            "k_run": "21/2",
        }

    def test_render_is_marker_idempotent_and_uses_canonical_policy(self):
        inventory = updater.render_inventory(self.counts())
        original = (
            "before\n" + updater.START + "\nold\n"
            + updater.END + "\nafter\n")
        first = updater.replace_inventory(original, inventory)
        second = updater.replace_inventory(first, inventory)
        self.assertEqual(first, second)
        self.assertIn("331", first)
        self.assertIn("21/2", first)
        self.assertIn("59", first)

    def test_wrong_k_or_schedule_fails_closed(self):
        for key, value in (("k_nodes", 57), ("k_run", "17/2")):
            counts = copy.deepcopy(self.counts())
            counts[key] = value
            with self.subTest(key=key):
                with self.assertRaisesRegex(ValueError, "requires"):
                    updater.render_inventory(counts)

    def test_duplicate_or_inverted_markers_fail(self):
        inventory = updater.render_inventory(self.counts())
        with self.assertRaisesRegex(ValueError, "exactly one"):
            updater.replace_inventory(updater.START * 2 + updater.END, inventory)
        with self.assertRaisesRegex(ValueError, "inverted"):
            updater.replace_inventory(
                updater.END + "\n" + updater.START, inventory)


if __name__ == "__main__":
    unittest.main()
