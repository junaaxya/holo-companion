from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SttAdmissionDiagnostics:
    active_inferences: int = 0
    cancelled_before_start: int = 0
    stale_results_discarded: int = 0


def bump_stt_cancel_diagnostic(diag: SttAdmissionDiagnostics, *, stt_ran: bool) -> SttAdmissionDiagnostics:
    if stt_ran:
        return SttAdmissionDiagnostics(
            diag.active_inferences,
            diag.cancelled_before_start,
            diag.stale_results_discarded + 1,
        )
    return SttAdmissionDiagnostics(
        diag.active_inferences,
        diag.cancelled_before_start + 1,
        diag.stale_results_discarded,
    )
