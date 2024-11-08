import { formStore } from "@/stores/FormStore";
import { observer } from "mobx-react";
import React from "react";

export const SelectedImage: React.FC = observer(() => {
  return (
    <div
      style={{
        display: "flex",
        maxWidth: "400px",
        maxHeight: "400px",
        alignItems: "center",
      }}
    >
      {formStore.uploadedImage && (
        <img
          src={formStore.uploadedImage}
          style={{
            width: "100%",
            height: "100%",
            maxHeight: "400px",
            objectFit: "contain",
            borderRadius: "8px",
          }}
        />
      )}
    </div>
  );
});
