import { observer } from "mobx-react";
import React from "react";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { ISimmilarImage } from "@/types/SimmilarImage";

interface ResponseCardProps {
  simmilarImage: ISimmilarImage;
}

export const ResponseCard: React.FC<ResponseCardProps> = observer(
  ({ simmilarImage }) => {
    return (
      <Card
        style={{
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
        }}
      >
        <CardHeader>
          <CardTitle>
            {`Distance: ${simmilarImage.distance.toFixed(4)}`}
          </CardTitle>
          <CardDescription>
            {`Certainty: ${simmilarImage.certainty.toFixed(4)}`}
            {` Cluster: ${simmilarImage.cluster}`}
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div
            style={{
              width: "40vw",
              // minWidth: "200px",
              height: "fit-content",
              borderRadius: "8px",
            }}
          >
            <img
              src={simmilarImage.image}
              style={{
                width: "100%",
                height: "100%",
                objectFit: "contain",
                borderRadius: "8px",
              }}
            ></img>
          </div>
        </CardContent>
      </Card>
    );
  }
);
