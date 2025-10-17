package com.khanhlms.medical_store.dtos.response.categories;

import lombok.*;
import lombok.experimental.FieldDefaults;
import org.springframework.web.multipart.MultipartFile;

@NoArgsConstructor
@AllArgsConstructor
@Setter
@Getter
@Builder
@FieldDefaults(level = AccessLevel.PRIVATE)
public class CategoryResponse {
    String id;
    String name;
    String description;
    String thumbnailUrl;
    Integer position;
    Boolean active;
}
