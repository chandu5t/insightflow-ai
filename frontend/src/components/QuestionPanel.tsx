import { useRef, useState, type FormEvent } from "react";
import { askQuestion } from "../services/api";
import type { QueryResponse } from "../types/query";
import ErrorMessage from "./ErrorMessage";
import QueryResult from "./QueryResult";

// These are only question texts. Every answer comes from the backend.
const EXAMPLE_QUESTIONS = [
  "What is the total revenue?",
  "Which region generated the highest revenue?",
  "What is the average unit price?",
  "What is the maximum quantity?",
  "Show revenue by region.",
  "Which products have the highest revenue?",
  "What is revenue?",
];

type QueryState =
  | { phase: "idle" }
  | { phase: "loading"; question: string }
  | { phase: "done"; response: QueryResponse }
  | { phase: "error"; question: string; error: Error };

type QuestionPanelProps = {
  datasetId: string;
};

function QuestionPanel({ datasetId }: QuestionPanelProps) {
  const [text, setText] = useState("");
  const [state, setState] = useState<QueryState>({ phase: "idle" });
  // Counts requests, so an old answer can never replace a newer one.
  const latestRequest = useRef(0);

  async function ask(question: string) {
    const trimmed = question.trim();
    if (!trimmed) {
      setState({
        phase: "error",
        question: "",
        error: new Error("Please type a question first."),
      });
      return;
    }

    latestRequest.current += 1;
    const requestId = latestRequest.current;
    setState({ phase: "loading", question: trimmed });

    try {
      const response = await askQuestion(datasetId, trimmed);
      if (latestRequest.current === requestId) {
        setState({ phase: "done", response });
      }
    } catch (error: unknown) {
      if (latestRequest.current === requestId) {
        setState({
          phase: "error",
          question: trimmed,
          error:
            error instanceof Error
              ? error
              : new Error("An unknown error occurred."),
        });
      }
    }
  }

  function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    void ask(text);
  }

  function handleExample(question: string) {
    setText(question);
    void ask(question);
  }

  const isLoading = state.phase === "loading";

  return (
    <div>
      <form className="question-form" onSubmit={handleSubmit}>
        <input
          type="text"
          value={text}
          onChange={(event) => setText(event.target.value)}
          placeholder="Ask a question about this dataset"
          disabled={isLoading}
          aria-label="Your question"
        />
        <button type="submit" disabled={isLoading}>
          {isLoading ? "Asking..." : "Ask"}
        </button>
      </form>

      <p className="hint">Try an example. Not every question works for every dataset.</p>
      <div className="example-list">
        {EXAMPLE_QUESTIONS.map((example) => (
          <button
            key={example}
            type="button"
            className="secondary-button example-button"
            disabled={isLoading}
            onClick={() => handleExample(example)}
          >
            {example}
          </button>
        ))}
      </div>

      {state.phase === "loading" && (
        <p className="status status-loading">Analysing: {state.question}</p>
      )}

      {state.phase === "error" && (
        <div>
          <ErrorMessage error={state.error} />
          {state.question && (
            <button type="button" onClick={() => void ask(state.question)}>
              Retry
            </button>
          )}
        </div>
      )}

      {state.phase === "done" && <QueryResult response={state.response} />}
    </div>
  );
}

export default QuestionPanel;