package com.khanhlms.medical_store.dtos.requests.categories;

import lombok.*;
import lombok.experimental.FieldDefaults;
import org.springframework.web.multipart.MultipartFile;

@NoArgsConstructor
@AllArgsConstructor
@Setter
@Getter
@Builder
@FieldDefaults(level = AccessLevel.PRIVATE)
public class CreateCategoryRequest {
    String name;
    String description;
    MultipartFile image;
}
