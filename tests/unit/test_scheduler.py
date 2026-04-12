from kvcore.api import Request
from kvcore.scheduler import Scheduler


def test_scheduler_uses_waiting_and_running_queues() -> None:
    scheduler = Scheduler()
    scheduler.add_request(Request(prompt="hello", request_id="req-1"))

    prefill_batch = scheduler.schedule()

    assert prefill_batch is not None
    assert prefill_batch.mode == "prefill"
    assert len(scheduler.waiting) == 0
    assert len(scheduler.running) == 1
