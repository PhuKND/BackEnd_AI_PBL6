package com.khanhlms.medical_store.dtos.products.requests;

import lombok.*;
import lombok.experimental.FieldDefaults;
import org.springframework.web.multipart.MultipartFile;

import java.util.Date;
import java.util.List;

@NoArgsConstructor
@AllArgsConstructor
@Setter
@Getter
@Builder
@FieldDefaults(level = AccessLevel.PRIVATE)
public class IngredientRequest {
    String name;
    Double amount;
    String unit;
    String description;
}
