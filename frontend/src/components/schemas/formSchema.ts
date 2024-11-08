const MAX_UPLOAD_SIZE = 1024 * 1024 * 10; // 3MB

import { z } from "zod";
export const formSchema = z.object({
  image: z
    .instanceof(FileList)
    .refine((file) => {
      return file.length > 0;
    }, "File is required")
    .refine((file) => {
      return !file.length || file[0].size <= MAX_UPLOAD_SIZE;
    }, "File size must be less than 3MB"),
});
