import { formStore } from "@/stores/FormStore";
import { observer } from "mobx-react";
import React from "react";
import { ResponseCard } from "./ResponseCard";

export const ResponseCards: React.FC = observer(() => {
  return (
    <>
      {formStore.response.length > 0 && <h1>Results:</h1>}

      <div
        style={{
          display: "flex",
          flexDirection: "row",
          gap: "32px",
          flexWrap: "wrap",
          justifyContent: "center",
        }}
      >
        {formStore.response.map((simmilarImage, idx) => (
          <ResponseCard key={idx} simmilarImage={simmilarImage} />
        ))}
      </div>
    </>
  );
});
