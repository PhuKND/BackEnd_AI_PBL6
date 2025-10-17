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
public class CreateProductRequest {
    String name;
    String description;
    Double price;
    String currency;
    Integer quantity;
    Boolean precription;
    String usage; // cách sử dụng
    String benefit; // lợi ích
    String sideEffect; // tác dụng phụ
    String note;
    String preserve;
    Date productDate;
    Date expirationDate;
    String manufacturerId;
    String categoryId;
    String unit;
    List<MultipartFile> images;
}
