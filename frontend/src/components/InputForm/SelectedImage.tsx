import { formStore } from "@/stores/FormStore";
import { observer } from "mobx-react";
import React from "react";

export const SelectedImage: React.FC = observer(() => {
  return (
    <div style={{ width: "400px", height: "400px" }}>
      {formStore.uploadedImage && (
        <img
          src={formStore.uploadedImage}
          style={{
            width: "100%",
            height: "100%",
            objectFit: "contain",
            borderRadius: "8px",
          }}
        />
      )}
    </div>
  );
});
