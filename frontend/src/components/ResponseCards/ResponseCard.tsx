import { observer } from "mobx-react";
import React from "react";
import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
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
      <Card>
        <CardHeader>
          <CardTitle>
            {`Distance: ${simmilarImage.distance.toFixed(4)}`}
          </CardTitle>
          <CardDescription>
            {`Certainty: ${simmilarImage.certainty.toFixed(4)}`}
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div style={{ width: "400px", height: "400px", borderRadius: "8px" }}>
            <img
              src={`data:image/png;base64,${simmilarImage.image}`}
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
