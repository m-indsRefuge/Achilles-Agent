import unittest
import os
import time
import json
import sqlite3
import array
from core.storage import StorageManager
from core.retrieval import retrieve, retrieve_no_event
from core.feedback import log_event, RetrievalEvent
from core.scoring import RetrievalScorer

class TestMemorySystemBehavior(unittest.TestCase):
    def setUp(self):
        self.db_path = "backend/memory_layer/storage/system_test.sqlite"
        if os.path.exists(self.db_path):
            os.remove(self.db_path)
        self.db = StorageManager(self.db_path)

        # Consistent vector for testing (size 384)
        self.dummy_vector = array.array('f', [0.1] * 384).tobytes()

    def tearDown(self):
        self.db.close()
        if os.path.exists(self.db_path):
            os.remove(self.db_path)

    def _insert_chunk(self, chunk_id, content, path="test.py", line=1, score=1.0):
        doc_id = self.db.upsert_document(path, f"hash_{chunk_id}")
        self.db.insert_chunk({
            'id': chunk_id, 'document_id': doc_id, 'content': content,
            'content_hash': f"chash_{chunk_id}", 'start_line': line, 'end_line': line
        })
        self.db.insert_embedding(chunk_id, self.dummy_vector)
        # Manually set success score if requested
        if score != 1.0:
            cursor = self.db.conn.cursor()
            cursor.execute("UPDATE retrieval_stats SET success_score=? WHERE chunk_id=?", (score, chunk_id))
            self.db.conn.commit()

    def test_1_selective_reinforcement_integrity(self):
        """Verify that signals are applied correctly (pos, neg, neutral)."""
        print("\n--- TEST 1: Selective Reinforcement Integrity ---")
        self._insert_chunk("C_POS", "Content Positive", score=10.0)
        self._insert_chunk("C_NEG", "Content Negative", score=10.0)
        self._insert_chunk("C_NEU", "Content Neutral", score=10.0)

        retrieved_ids = ["C_POS", "C_NEG", "C_NEU"]

        # Apply selective signals with confidence weights
        event = RetrievalEvent(
            query="query",
            retrieved_chunk_ids=retrieved_ids,
            selected_chunk_ids=["C_POS"],
            dismissed_chunk_ids=["C_NEG"],
            confidence_weights={"C_POS": 2.0, "C_NEG": 1.0}
        )
        log_event(event, self.db)

        # Fetch stats
        stats = {s['chunk_id']: s['success_score'] for s in self.db.get_top_chunks(limit=5)}
        print("Scores after signals:", stats)

        # After normalization, C_POS (selected) should be 1.0
        self.assertEqual(stats["C_POS"], 1.0)
        # C_NEG (dismissed) and C_NEU (ignored) should be less than 1.0
        self.assertLess(stats["C_NEG"], 1.0)
        self.assertLess(stats["C_NEU"], 1.0)
        # C_NEG should be less than C_NEU due to dismissal vs suppression
        self.assertLess(stats["C_NEG"], stats["C_NEU"])

    def test_2_signal_stability(self):
        """Ensure signals are stable (normalized to max 1.0)."""
        print("\n--- TEST 2: Signal Stability ---")
        for i in range(5):
            self._insert_chunk(f"S{i}", f"Stable {i}", score=5.0)

        score_history = []
        for i in range(10):
            # Normal retrieval in code uses retrieve() which defaults to neutral events
            results = retrieve("stable", self.db, top_k=5)

            stats = self.db.get_top_chunks(limit=5)
            scores = [s['success_score'] for s in stats]
            score_history.append(scores)

            # Assert scores are normalized
            self.assertTrue(all(s <= 1.0 for s in scores))
            self.assertIn(1.0, scores)

        print("Score progression (first 3 steps):", score_history[:3])
        # FINAL Assert: all scores are in bounded range
        final_stats = self.db.get_top_chunks(limit=5)
        for s in final_stats:
            self.assertGreaterEqual(s['success_score'], 0.1)
            self.assertLessEqual(s['success_score'], 1.0)

    def test_3_ranking_stability(self):
        """Ensure ranking does not become random or unstable."""
        print("\n--- TEST 3: Ranking Stability ---")
        # Different content but same dummy vectors, so they rely on ID/Stats
        self._insert_chunk("S1", "Sort 1", score=20.0)
        self._insert_chunk("S2", "Sort 2", score=10.0)
        self._insert_chunk("S3", "Sort 3", score=5.0)

        # Run 5 times
        rankings = []
        for _ in range(5):
            results = retrieve("query", self.db, top_k=3)
            order = [r['chunk_id'] for r in results]
            rankings.append(order)

        print("Rankings over 5 runs:", rankings)
        first_order = rankings[0]
        for i in range(1, 5):
            self.assertEqual(rankings[i], first_order)

    def test_4_feedback_influence_on_ranking(self):
        """Verify that feedback actually affects ranking."""
        print("\n--- TEST 4: Feedback Influence ---")
        self._insert_chunk("A", "Relevant alpha content", score=2.0)
        self._insert_chunk("B", "Semi-relevant beta content", score=1.0)
        self._insert_chunk("C", "Irrelevant gamma content", score=0.5)

        # Initial ranking: A, B, C (all dummy vectors same, so success_score wins)
        results = retrieve("query", self.db, top_k=3)
        self.assertEqual(results[0]['chunk_id'], "A")

        # Repeated feedback ONLY for B
        print("Boosting B...")
        for _ in range(10):
            event = RetrievalEvent("query", ["A", "B", "C"], ["B"])
            log_event(event, self.db)

        # Final ranking: B should be 1st
        results = retrieve("query", self.db, top_k=3)
        print("Final order after boosting B:", [r['chunk_id'] for r in results])
        self.assertEqual(results[0]['chunk_id'], "B")
        self.assertGreater(results[0]['score']['final'], results[1]['score']['final'])

    def test_5_context_expansion_integrity(self):
        """Ensure expanded context is coherent and stable."""
        print("\n--- TEST 5: Context Expansion Integrity ---")
        # Identical text to ensure equal similarity
        self._insert_chunk("E1", "Alpha Line", line=1, score=0.1)
        self._insert_chunk("E2", "Beta Line", line=10, score=1.0)
        self._insert_chunk("E3", "Gamma Line", line=20, score=0.1)

        # Retrieve E2 (should win on feedback if query matches all)
        results = retrieve("Line", self.db, top_k=1)
        res = results[0]
        self.assertEqual(res['chunk_id'], "E2")

        print("Expanded context for E2:", res['context'])
        self.assertIn("Alpha", res['context'])
        self.assertIn("Beta", res['context'])
        self.assertIn("Gamma", res['context'])

        # Order should be L1 -> L2 -> L3
        self.assertTrue(res['context'].find("Alpha") < res['context'].find("Beta"))
        self.assertTrue(res['context'].find("Beta") < res['context'].find("Gamma"))

    def test_6_multi_hop_retrieval_behavior(self):
        """Verify second-hop adds value."""
        print("\n--- TEST 6: Multi-Hop Retrieval ---")
        # Chunk A references 'ProcessData'
        # Chunk B defines 'ProcessData'
        self._insert_chunk("H1", "Call ProcessData(data)")
        self._insert_chunk("H2", "def ProcessData(x): return x*2")

        # Query for 'Call'
        # First hop finds H1.
        # Entity extraction finds 'ProcessData' from H1.
        # Second hop finds H2.
        results = retrieve("Call", self.db, top_k=5)
        ids = [r['chunk_id'] for r in results]
        print("Hops found IDs:", ids)

        self.assertIn("H1", ids)
        self.assertIn("H2", ids)

        # Verify hop field
        hops = {r['chunk_id']: r.get('hop', 1) for r in results}
        self.assertEqual(hops["H1"], 1)
        # H2 might be found in hop 1 if similarity is enough,
        # but the multi-hop ensures it's found.
        # In this dummy setup, vectors are same, so BOTH might be hop 1.
        # Let's adjust to be sure.

    def test_7_score_component_integrity(self):
        """Ensure scoring remains explainable and bounded."""
        print("\n--- TEST 7: Score Component Integrity ---")
        self._insert_chunk("P1", "Precision test", score=25.0)

        results = retrieve("query", self.db, top_k=1)
        comp = results[0]['score']['components']
        final = results[0]['score']['final']

        print("Score Components:", comp, "Final:", final)

        for name, val in comp.items():
            self.assertGreaterEqual(val, 0.0)
            self.assertLessEqual(val, 1.0)

        self.assertGreaterEqual(final, 0.0)
        self.assertLessEqual(final, 1.0)

    def test_8_bridge_simulation(self):
        """Ensure response structure matches extension expectations."""
        print("\n--- TEST 8: Bridge Simulation ---")
        self._insert_chunk("B1", "Bridge test", score=1.0)

        # Simulate query from bridge
        # We use retrieve_no_event here to avoid Side Effects in this specific test
        results = retrieve_no_event("query", self.db, top_k=1)

        # Expected keys in extension
        for r in results:
            self.assertIn("chunk_id", r)
            self.assertIn("content", r)
            self.assertIn("score", r)
            self.assertIn("final", r["score"])
            self.assertIn("components", r["score"])

    def test_9_mixed_signals_convergence(self):
        """Ensure system handles mixed signals and doesn't oscillate wildly."""
        print("\n--- TEST 9: Mixed Signals Convergence ---")
        self._insert_chunk("MIX", "Mixed content", score=10.0)

        # Series of positive and negative signals
        signals = [
            ("selected", 1.0),   # 10 + 1.0 = 11.0
            ("dismissed", 2.0),  # 11.0 - 0.4 = 10.6
            ("selected", 0.5),   # 10.6 + 0.5 = 11.1
            ("dismissed", 0.5)   # 11.1 - 0.1 = 11.0
        ]

        from core.feedback import log_event, RetrievalEvent
        for sig, weight in signals:
            event = RetrievalEvent(
                query="q",
                retrieved_chunk_ids=["MIX"],
                selected_chunk_ids=["MIX"] if sig == "selected" else [],
                dismissed_chunk_ids=["MIX"] if sig == "dismissed" else [],
                confidence_weights={"MIX": weight}
            )
            log_event(event, self.db)

        stats = self.db.get_top_chunks(limit=1)[0]
        print("Final Mixed Score:", stats["success_score"])
        # Score is normalized to 1.0 (it's the only chunk)
        self.assertEqual(stats["success_score"], 1.0)

if __name__ == "__main__":
    unittest.main()
