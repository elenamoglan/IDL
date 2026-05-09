from transformers import pipeline


class Summarizer:
    def __init__(self, model_name="elena22222222/my-bart-summarizer"):
        self.model_name = model_name
        self.summarizer = pipeline("summarization", model=self.model_name)

    def summarize(self, text, max_length=150, min_length=40):
        """
        Generates a summary for the given text.
        If the text is too long, we might need to chunk it, but for simplicity,
        we'll summarize it as a single chunk if it's within the model's max tokens (usually 1024),
        or truncate it.
        """
        max_chunk_size = 3000
        chunks = [
            text[i : i + max_chunk_size] for i in range(0, len(text), max_chunk_size)
        ]

        summary_text = ""
        for chunk in chunks:
            if len(chunk) < 50:
                continue

            # calculate dynamic max_length if chunk is small
            current_max = min(max_length, len(chunk) // 2)
            current_min = min(min_length, current_max - 10)

            if current_max <= current_min:
                current_min = max(5, current_max - 5)

            try:
                print(f"Summarizing text")
                res = self.summarizer(
                    chunk,
                    max_new_tokens=current_max,
                    min_length=current_min,
                    do_sample=False,
                )
                summary_text += res[0]["summary_text"] + " "
            except Exception as e:
                print(f"Error summarizing chunk: {e}")

        return summary_text.strip()


if __name__ == "__main__":
    summarizer = Summarizer()
    text = "This is a very long text that needs to be summarized. " * 50
    print("Summary:", summarizer.summarize(text))
