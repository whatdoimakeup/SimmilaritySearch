import { makeAutoObservable } from "mobx";
// import { z } from "zod";
import axios from "axios";
import { z } from "zod";
import { formSchema } from "@/components/schemas/formSchema";
import { ISimmilarImage } from "@/types/SimmilarImage";
class FormStore {
  images: FileList | null = null;
  response: ISimmilarImage[] = [];
  constructor() {
    makeAutoObservable(this);
  }

  setImage(image: FileList | null) {
    this.images = image;
  }

  get uploadedImage() {
    if (!this.images) return null;
    return URL.createObjectURL(this.images[0]);
  }

  async submit(values: z.infer<typeof formSchema>) {
    const formData = new FormData();
    formData.append("image", values.image[0]);

    try {
      const response = await axios.post(
        "https://api.meiiiok.ru/api/search",
        formData,
        {
          headers: {
            "Content-Type": "multipart/form-data",
          },
        }
      );

      this.response = response.data;
      console.log(response.data);
    } catch (error) {
      console.error("There was an error uploading the image!", error);
    }
  }
}
export const formStore = new FormStore();
