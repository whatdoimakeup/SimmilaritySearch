import "./App.css";
import { InputForm } from "./components/InputForm/InputForm";
import { ResponseCards } from "./components/ResponseCards/ResponseCards";
import { Label } from "./components/ui/label";
import { Switch } from "./components/ui/switch";
import { observer } from "mobx-react";
import { formStore } from "./stores/FormStore";
const App = observer(() => {
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "32px" }}>
      <div className="flex flex-row gap-3 items-center w-full">
        <Switch
          checked={formStore.save}
          onCheckedChange={(checked) => formStore.setSave(checked)}
        />
        <Label>Save in database</Label>
      </div>

      <InputForm />

      <ResponseCards />
    </div>
  );
});
export default App;
