import json
import unittest
from pathlib import Path
from urllib.parse import parse_qs, urlparse


GRAPH_APP_DIR = Path(__file__).resolve().parents[1]
EVAL_DIR = GRAPH_APP_DIR / "eval"
VERIFIED_CIRCUITS_DIR = EVAL_DIR / "verified-circuits"
VERIFIED_CIRCUITS_PATH = VERIFIED_CIRCUITS_DIR / "verified-circuits.json"
VERIFIED_CIRCUITS_SCHEMA_PATH = VERIFIED_CIRCUITS_DIR / "verified-circuits.schema.json"


class VerifiedCircuitsTest(unittest.TestCase):
    def setUp(self):
        with VERIFIED_CIRCUITS_PATH.open() as f:
            self.data = json.load(f)

    def test_fixture_has_expected_top_level_contract(self):
        self.assertEqual(self.data["$schema"], "./verified-circuits.schema.json")
        self.assertEqual(self.data["schema_version"], 1)
        self.assertEqual(len(self.data["circuits"]), 15)

        orders = [circuit["order"] for circuit in self.data["circuits"]]
        ids = [circuit["id"] for circuit in self.data["circuits"]]
        self.assertEqual(orders, list(range(1, len(orders) + 1)))
        self.assertEqual(len(ids), len(set(ids)))

    def test_each_circuit_has_decoded_source_url_state(self):
        required_circuit_keys = {
            "id",
            "order",
            "label",
            "model_id",
            "source",
            "prompt",
            "target_token",
            "graph",
            "circuit",
            "view",
            "metadata",
        }

        for circuit in self.data["circuits"]:
            with self.subTest(circuit=circuit["id"]):
                self.assertEqual(set(circuit), required_circuit_keys)
                self.assertEqual(circuit["source"]["type"], "researcher_verified")
                self.assertTrue(circuit["prompt"])
                self.assertTrue(circuit["target_token"])

                parsed = urlparse(circuit["source"]["url"])
                query = parse_qs(parsed.query)
                path_parts = [part for part in parsed.path.split("/") if part]
                self.assertEqual(parsed.scheme, "https")
                self.assertEqual(parsed.netloc, "www.neuronpedia.org")
                self.assertEqual(path_parts, [circuit["model_id"], "graph"])
                self.assertEqual(query["slug"], [circuit["id"]])
                self.assertTrue(circuit["id"].endswith(circuit["graph"]["slug"]))

                pinned_ids = circuit["circuit"]["pinned_ids"]
                self.assertGreater(len(pinned_ids), 0)
                self.assertEqual(pinned_ids, query["pinnedIds"][0].split(","))

                for supernode in circuit["circuit"]["supernodes"]:
                    self.assertTrue(supernode["label"])
                    self.assertGreater(len(supernode["member_ids"]), 0)

                for clerp in circuit["circuit"]["clerps"]:
                    self.assertTrue(clerp["node_id"])
                    self.assertTrue(clerp["label"])

                for threshold in circuit["view"].values():
                    if threshold is not None:
                        self.assertGreaterEqual(threshold, 0)
                        self.assertLessEqual(threshold, 1)

    def test_known_circuit_values_are_preserved(self):
        first = self.data["circuits"][0]
        self.assertEqual(first["id"], "gemma-girls-are")
        self.assertEqual(first["prompt"], "The girls that the teacher sees")
        self.assertEqual(first["target_token"], "are")
        self.assertEqual(first["circuit"]["clicked_id"], "3_9864_3")
        self.assertEqual(first["circuit"]["supernodes"][0]["label"], "see/saw")
        self.assertEqual(
            first["circuit"]["supernodes"][0]["member_ids"],
            ["15_233_6", "6_11265_6", "3_6616_6"],
        )

    def test_schema_file_documents_the_fixture_contract(self):
        with VERIFIED_CIRCUITS_SCHEMA_PATH.open() as f:
            schema = json.load(f)

        self.assertEqual(schema["$schema"], "https://json-schema.org/draft/2020-12/schema")
        self.assertEqual(schema["properties"]["schema_version"]["const"], 1)
        self.assertIn("circuit", schema["$defs"])
        self.assertIn("supernode", schema["$defs"])
        self.assertIn("clerp", schema["$defs"])


if __name__ == "__main__":
    unittest.main()
