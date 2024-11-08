import { flow, makeAutoObservable } from "mobx";
// import { z } from "zod";
import axios, { AxiosResponse } from "axios";
import { z } from "zod";
import { formSchema } from "@/components/schemas/formSchema";
import { ISimmilarImage } from "@/types/SimmilarImage";
class FormStore {
  images: FileList | null = null;
  response: ISimmilarImage[] = [];
  isLoading = false;
  constructor() {
    makeAutoObservable(this, {
      submit: flow,
    });
  }

  setImage(image: FileList | null) {
    this.images = image;
  }

  get uploadedImage() {
    if (!this.images) return null;
    return URL.createObjectURL(this.images[0]);
  }

  *submit(values: z.infer<typeof formSchema>) {
    this.isLoading = true;
    const formData = new FormData();
    formData.append("image", values.image[0]);

    try {
      const response: AxiosResponse = yield axios.post(
        "http://localhost:8000/api/search",
        formData,
        {
          headers: {
            "Content-Type": "multipart/form-data",
          },
        }
      );

      this.response = response.data;
    } catch (error) {
      console.error("There was an error uploading the image!", error);
    } finally {
      this.isLoading = false;
    }
  }
}
export const formStore = new FormStore();
