from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
import os
import random

def optimize_cipher_tokens(toxic_texts, num_iterations):
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_name = "toxic-bert"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
    model.eval()

    def calculate_toxicity_score(input_text):
        inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        with torch.no_grad():
            logits = model(**inputs).logits
            probabilities = F.softmax(logits, dim=1)
            toxicity_score = probabilities[:, 0]
        return toxicity_score.item()

    emoji_chars = (
        "😀😃😄😁😆😅😂🤣😊😇🙂🙃😉😌😍🥰😘😗😙😚😋😜😝😛🤑🤗🤔🤐🤨😶😏😒😞😔😟😕🙁☹️😣😖😫😩🥺"
        "😤😠😡🤬😳🥵🥶😱😨😰😥😓🤯😢😭😤😠😡🤬🤢🤮🤧😷🤒🤕🤑😎🥳🤠🥸🤓🧐😶‍🌫️😬🤥😵‍💫😵🤯🤡"
        "👹👺💀☠️👻👽👾🤖💩🔥💫🌟✨💥💢💤💦💨🕳️🛑🕵️‍♂️🕵️‍♀️👨‍🚀👩‍🚀🧑‍🚀👨‍⚖️👩‍⚖️🤴👸👳‍♂️👳‍♀️🧕"
        "👮‍♂️👮‍♀️👷‍♂️👷‍♀️💂‍♂️💂‍♀️👨‍🏫👩‍🏫👨‍💻👩‍💻🕺💃🕴️👯‍♂️👯‍♀️🎅🤶🧑‍🎄👼🤰🤱👶👦👧👨👩👴👵"
        "🐶🐱🐭🐹🐰🦊🦝🐻🐼🐨🐯🦁🐮🐷🐽🐸🐵🐔🐧🐦🐤🦆🦅🦉🦇🐺🐗🐴🦄🐝🐛🦋🐌🐞🐜🦗🕷️🕸️🦂🐢🐍🦎"
        "🦖🦕🐙🦑🦐🦞🐠🐟🐡🐬🦭🐳🐋🦈🐊🐅🐆🦓🦍🦧🐘🦛🦏🐪🐫🦒🦘🦬🐄🐎🐖🐏🐑🦙🐐🦌🐕‍🦺🐩🐈🐓🦃🕊️"
    )
    additional_special_chars = (
        "€£¥₹©®™✓✔✕✖✗✘⚠️⚡☔✨❌❗❓❕❔➡️⬅️⬆️⬇️➕➖➗✱✲"
        "🔴🟠🟡🟢🔵🟣🟤⚫⚪🟥🟧🟨🟩🟦🟪🟫🔺🔻🔸🔹🔶🔷🔳🔲🧡💛💚💙💜🤎🖤🤍♥️♠️♣️♦️♤♧♢♣️♔♕♚♛🎲🎯🧩"
        "💎⚜️🎖️🏆🎁🎈🎉🎊🎇🎆✨🏅🎨🎵🎶🎧🎤🎬🎸🎻🎺🥁🎷🎹💿📀🎞️🎥📷📸📹📼🖼️🎮🕹️🗡️⚔️🔪🪓🔫"
    )

    vocab_tokens = list(tokenizer.get_vocab().keys())
    char_tokens = tokenizer.tokenize(emoji_chars + additional_special_chars)
    initial_token_set = vocab_tokens + char_tokens
    random.shuffle(initial_token_set)

    initial_text = ["!"] * 20
    initial_text_tokens = tokenizer.convert_tokens_to_ids(initial_text)

    best_initial_tokens = initial_text_tokens.copy()
    best_overall_score = float('inf')

    for iteration in range(num_iterations):
        for toxic_text in toxic_texts:
            modified_toxic_text = ' '.join(
                [f"{tokenizer.decode(best_initial_tokens)} {word} {tokenizer.decode(best_initial_tokens)}" for word in
                 toxic_text.split()]
            )

            current_score = calculate_toxicity_score(modified_toxic_text)

            for position in range(len(best_initial_tokens)):
                best_token = best_initial_tokens[position]
                best_local_score = current_score

                candidates = []
                for _ in range(10):
                    random_token = tokenizer.convert_tokens_to_ids(random.choice(initial_token_set))
                    candidates.append(random_token)

                candidate_scores = []
                for candidate_token in candidates:
                    test_initial_tokens = best_initial_tokens[:position] + [candidate_token] + best_initial_tokens[
                                                                                               position + 1:]
                    modified_text = ' '.join(
                        [f"{tokenizer.decode(test_initial_tokens)} {word} {tokenizer.decode(test_initial_tokens)}" for
                         word in toxic_text.split()]
                    )
                    new_score = calculate_toxicity_score(modified_text)
                    candidate_scores.append((candidate_token, new_score))

                top_candidates = sorted(candidate_scores, key=lambda x: x[1])[:5]

                best_candidate_token, best_candidate_score = top_candidates[0]

                if best_candidate_score < best_local_score:
                    best_local_score = best_candidate_score
                    best_token = best_candidate_token

                best_initial_tokens[position] = best_token
                current_score = best_local_score

            if current_score < best_overall_score:
                best_overall_score = current_score

    final_modified_text = ' '.join(
        [f"{tokenizer.decode(best_initial_tokens)} {word} {tokenizer.decode(best_initial_tokens)}" for word in
         toxic_text.split()]
    )
    final_score = calculate_toxicity_score(final_modified_text)
    return final_modified_text

# Example usage
toxic_texts = ["stop being criminals and you wont get shot", "your actions have consequences"]
num_iterations = 10
optimized_tokens = optimize_cipher_tokens(toxic_texts, num_iterations)
