import { useState } from "react";
import "./App.css";

function App() {
  const [text, setText] = useState("");
  const [genres, setGenres] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";
  const handleSubmit = async () => {
    if (!text.trim()) return;

    setLoading(true);
    setError(null);
    setGenres(null);

    try {
      const response = await fetch(`${API_URL}/predict`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ text }),
      });

      if (!response.ok) {
        throw new Error("Failed to fetch prediction");
      }

      const data = await response.json();
      setGenres(data.predicted_genres);
    } catch (err) {
      console.error(err);
      setError("Something went wrong. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <>
      <h1>Genre Classifier</h1>
      <div className="card">
        <textarea
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder="Paste your book description or text here..."
          disabled={loading}
        />
        <button onClick={handleSubmit} disabled={loading || !text.trim()}>
          {loading ? "Analyzing..." : "Predict Genre"}
        </button>

        {error && <div className="error">{error}</div>}

        {genres && (
          <div className="result">
            <h2>Predicted Genres</h2>
            <div className="tags">
              {genres.map((genre, index) => (
                <span key={index} className="tag">
                  {genre}
                </span>
              ))}
            </div>
          </div>
        )}
      </div>
    </>
  );
}

export default App;
