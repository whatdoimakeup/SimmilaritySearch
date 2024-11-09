import { zodResolver } from "@hookform/resolvers/zod";
import { useForm } from "react-hook-form";
import { z } from "zod";

import { Button } from "@/components/ui/button";
import {
  Form,
  FormControl,
  FormDescription,
  FormField,
  FormItem,
  FormLabel,
  FormMessage,
} from "@/components/ui/form";
import { Input } from "@/components/ui/input";
import { observer } from "mobx-react";
import { formStore } from "@/stores/FormStore";
import { SelectedImage } from "./SelectedImage";

import { formSchema } from "../schemas/formSchema";

export const InputForm: React.FC = observer(() => {
  const form = useForm<z.infer<typeof formSchema>>({
    resolver: zodResolver(formSchema),
  });

  // 2. Define a submit handler.
  async function onSubmit(values: z.infer<typeof formSchema>) {
    formStore.submit(values);
  }
  const imageRef = form.register("image");

  return (
    <div
      style={{ display: "flex", flexDirection: "column", alignItems: "center" }}
    >
      <SelectedImage />

      <Form {...form}>
        <form onSubmit={form.handleSubmit(onSubmit)} className="space-y-8">
          <FormField
            control={form.control}
            name="image"
            render={({ field }) => (
              <FormItem>
                <FormLabel>Image</FormLabel>
                <FormControl>
                  <Input
                    placeholder="shadcn"
                    type="file"
                    accept="image/jpg,image/jpeg,image/png"
                    {...imageRef}
                    onChange={(event) => {
                      formStore.setImage(event.target.files);
                      field.onChange(event.target?.files ?? undefined);
                    }}
                  />
                </FormControl>
                <FormDescription>
                  Image to find silmmilarities with
                </FormDescription>
                <FormMessage />
              </FormItem>
            )}
          />
          <Button disabled={formStore.isLoading} type="submit">
            {formStore.isLoading ? "Searching..." : "Find simmilar images"}
          </Button>
        </form>
      </Form>
    </div>
  );
});
