import "./App.css";
import { InputForm } from "./components/InputForm/InputForm";
import { ResponseCards } from "./components/ResponseCards/ResponseCards";
import { ThemeProvider } from "./components/ThemeProvider";

// 1. Define your form.

function App() {
  return (
    // <ThemeProvider value={{ theme: "dark" }}>
    <div style={{ display: "flex", flexDirection: "column", gap: "32px" }}>
      <InputForm />

      <ResponseCards />
    </div>
    // </ThemeProvider>
  );
}

export default App;
