"""
Batched GPU inference server for parallel MCTS.

Collects evaluation requests from multiple self-play workers via a shared
queue, batches them, runs a single GPU forward pass, and sends results back.

This is the key to GPU utilization — individual per-leaf inference calls
would waste 99%+ of the 4090's throughput.
"""

import torch
import numpy as np
from alphazero.model import collate_states, uncollate_policy


class InferenceServer:
    """
    GPU inference server that batches requests from multiple workers.

    Usage (in a dedicated process):
        server = InferenceServer(model, device, batch_size=32, timeout=0.005)
        server.run(request_queue, response_queues)

    Workers send: (worker_id, encoded_state, valid_moves)
    Workers receive: (policy_array, value_float)
    """

    def __init__(self, model, device, batch_size=32, timeout=0.005):
        """
        Args:
            model: ResNet model (already on device).
            device: torch device.
            batch_size: max batch size before forcing a forward pass.
            timeout: seconds to wait for more requests before flushing.
        """
        self.model = model
        self.device = device
        self.batch_size = batch_size
        self.timeout = timeout

    @torch.no_grad()
    def process_batch(self, requests):
        """
        Run inference on a batch of requests.

        Args:
            requests: list of (worker_id, encoded_state, valid_moves).

        Returns:
            list of (worker_id, policy_np, value_float).
        """
        if not requests:
            return []

        worker_ids = [r[0] for r in requests]
        encoded_states = [r[1] for r in requests]
        valid_moves_list = [r[2] for r in requests]

        # Collate to padded batch
        batch_tensor, batch_valid, orig_sizes = collate_states(
            encoded_states, valid_moves_list
        )
        batch_tensor = batch_tensor.to(self.device)
        batch_valid = batch_valid.to(self.device)

        # Forward pass
        policy_logits, values = self.model(batch_tensor, batch_valid)

        # Softmax on policy (the model returns raw logits when mask is applied)
        policy_probs = torch.softmax(policy_logits, dim=1)

        # Uncollate policies back to original sizes
        policies = uncollate_policy(policy_probs, orig_sizes)
        values_np = values.cpu().numpy().flatten()

        results = []
        for i, wid in enumerate(worker_ids):
            results.append((wid, policies[i], float(values_np[i])))

        return results

    def run(self, request_queue, response_queues, stop_event):
        """
        Main server loop. Collects requests, batches, and dispatches results.

        Args:
            request_queue: multiprocessing.Queue — workers send requests here.
            response_queues: dict of {worker_id: multiprocessing.Queue}.
            stop_event: multiprocessing.Event — set to stop the server.
        """
        import queue

        self.model.eval()
        pending = []

        while not stop_event.is_set():
            # Collect requests
            try:
                req = request_queue.get(timeout=self.timeout)
                pending.append(req)
            except queue.Empty:
                pass

            # Process batch when full or timeout
            if len(pending) >= self.batch_size or (
                pending and len(pending) > 0
            ):
                results = self.process_batch(pending)
                for wid, policy, value in results:
                    if wid in response_queues:
                        response_queues[wid].put((policy, value))
                pending = []

        # Flush remaining
        if pending:
            results = self.process_batch(pending)
            for wid, policy, value in results:
                if wid in response_queues:
                    response_queues[wid].put((policy, value))
