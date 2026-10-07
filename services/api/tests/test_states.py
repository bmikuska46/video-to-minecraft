from __future__ import annotations

import unittest

from app.states import (
    InvalidStateTransition,
    ScanStatus,
    require_scan_transition,
)


class ScanStateMachineTests(unittest.TestCase):
    def test_happy_path_is_accepted(self):
        states = [
            ScanStatus.CREATED,
            ScanStatus.UPLOADING,
            ScanStatus.UPLOADED,
            ScanStatus.QUEUED,
            ScanStatus.EXTRACTING_FRAMES,
            ScanStatus.SPARSE_RECONSTRUCTION,
            ScanStatus.DENSE_RECONSTRUCTION,
            ScanStatus.FILTERING,
            ScanStatus.PREVIEW_READY,
        ]
        for current, target in zip(states, states[1:]):
            require_scan_transition(current, target)

    def test_skipping_or_leaving_terminal_state_is_rejected(self):
        with self.assertRaises(InvalidStateTransition):
            require_scan_transition(ScanStatus.CREATED, ScanStatus.QUEUED)
        with self.assertRaises(InvalidStateTransition):
            require_scan_transition(ScanStatus.PREVIEW_READY, ScanStatus.FILTERING)

    def test_every_processing_state_can_fail(self):
        for current in ScanStatus:
            if current in {ScanStatus.PREVIEW_READY, ScanStatus.FAILED}:
                continue
            require_scan_transition(current, ScanStatus.FAILED)


if __name__ == "__main__":
    unittest.main()
