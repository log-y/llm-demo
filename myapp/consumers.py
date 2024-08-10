# your_app_name/consumers.py

import json
from channels.generic.websocket import AsyncWebsocketConsumer

class GenerateTextConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        # Accept the WebSocket connection
        await self.accept()

    async def disconnect(self, close_code):
        # Handle disconnection (optional)
        pass

    async def receive(self, text_data):
        data = json.loads(text_data)
        input_text = data.get('input_text', '')
        num_sentences = data.get('num_sentences', 2)
        num_predictions = data.get('num_predictions', 10)

        # Call your generate_text function
        results = await self.generate_text(input_text, num_sentences, num_predictions)

        # Send the results to the WebSocket
        await self.send(text_data=json.dumps(results))

    async def generate_text(self, input_text, num_sentences, num_predictions):
        # Import necessary modules and set up your environment
        import torch
        import os
        from .models.model import GPT
        from .models.model import GPTConfig
        import tiktoken
        from torch.nn import functional as F
        import random
        import time

        CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), 'models', 'model.pt')
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=torch.device('cpu'))

        model = GPT(GPTConfig(vocab_size=50257))
        model.load_state_dict(checkpoint)

        model.eval()
        enc = tiktoken.get_encoding("gpt2")
        device="cpu"

        # Define the generate_text logic
        start_time = time.time()
        tokens = enc.encode(input_text)
        initial_token_ids = tokens
        tokens = torch.tensor(tokens, dtype=torch.long)
        tokens = tokens.unsqueeze(0).repeat(1, 1)
        xgen = tokens.to(device)
        sample_rng = torch.Generator(device=device)
        sample_rng.manual_seed(random.randint(20, 900))
        generated_texts = []
        curr_sentences = 0
        all_predictions = []

        while curr_sentences < num_sentences:
            with torch.no_grad():
                logits, _ = model(xgen)
            logits = logits[:, -1, :]
            probs = torch.nn.functional.softmax(logits, dim=-1)
            topk_probs, topk_indices = torch.topk(probs, num_predictions, dim=-1)
            ix = torch.multinomial(topk_probs, 1, generator=sample_rng)
            xcol = torch.gather(topk_indices, -1, ix)
            xgen = torch.cat((xgen, xcol), dim=1)

            top_predictions = [(enc.decode_single_token_bytes(topk_indices[0, i].item()).decode('utf-8'), topk_probs[0, i].item()) for i in range(num_predictions)]
            all_predictions.append(top_predictions)

            endsentencetokens = [13, 30, 0]
            startdelimiter = [4, 330, 58, 90]
            enddelimiter = [883, 330, 60, 92]
            stack = []
            newtoken = xcol[0, 0].item()
            if newtoken in startdelimiter:
                stack.append(newtoken)
                continue
            if newtoken in enddelimiter and len(stack) > 0:
                if (newtoken == 883 and stack[-1] == 4) or \
                   (newtoken == 330 and stack[-1] == 330) or \
                   (newtoken == 60 and stack[-1] == 58) or \
                   (newtoken == 92 and stack[-1] == 90):
                    stack = stack[:-1]
            if (newtoken in endsentencetokens and len(stack) == 0) or newtoken == 1210:
                curr_sentences += 1

        tokens = xgen[0].tolist()
        decoded = enc.decode(tokens)
        generated_texts.append(f"{decoded}")
        final_token_strings = [enc.decode_single_token_bytes(i).decode('utf-8') for i in tokens]
        end_time = time.time()
        total_time = end_time - start_time

        return {
            "response": "\n".join(generated_texts),
            "time": total_time,
            "predictions": all_predictions,
            "tokens": tokens,
            "initial_token_ids": initial_token_ids,
            "final_token_strings": final_token_strings
        }