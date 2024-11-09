import { flow, makeAutoObservable } from "mobx";
// import { z } from "zod";
import axios, { AxiosResponse } from "axios";
import { z } from "zod";
import { formSchema } from "@/components/schemas/formSchema";
import { ISimmilarImage } from "@/types/SimmilarImage";
class FormStore {
  images: FileList | null = null;
  response: ISimmilarImage[] = [];
  save = false;
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
    if (!this.images || !this.images[0]) return null;

    return URL.createObjectURL(this.images[0]);
  }

  setSave(save: boolean) {
    this.save = save;
  }

  *submit(values: z.infer<typeof formSchema>) {
    this.isLoading = true;
    const formData = new FormData();
    formData.append("image", values.image[0]);
    formData.append("save", String(this.save));

    try {
      const response: AxiosResponse = yield axios.post(
        "https://api.meiiiok.ru/api/search",
        // "http://127.0.0.1:8000/api/search",
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
      this.save = false;
    }
  }
}
export const formStore = new FormStore();
