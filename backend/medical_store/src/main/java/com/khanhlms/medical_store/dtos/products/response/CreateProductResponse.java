package com.khanhlms.medical_store.dtos.products.response;

import com.fasterxml.jackson.annotation.JsonFormat;
import com.khanhlms.medical_store.dtos.products.requests.IngredientRequest;
import org.springframework.web.multipart.MultipartFile;

import lombok.*;
import lombok.experimental.FieldDefaults;

import java.util.Date;
import java.util.List;

@NoArgsConstructor
@AllArgsConstructor
@Setter
@Getter
@Builder
@FieldDefaults(level = AccessLevel.PRIVATE)
public class CreateProductResponse {
    String id;
    String name;
    String description;
    Double originPrice;
    String currency;
    Boolean precription;
    String usage; // cách sử dụng
    String benefit; // lợi ích
    List<IngredientRequest> ingredients; // thanf phan
    String sideEffect; // tác dụng phụ
    String note;
    String preserve;
    Integer quantity;
    @JsonFormat( pattern = "dd/MM/yyyy")
    Date productDate;
    @JsonFormat(pattern = "dd/MM/yyyy")
    Date expirationDate;
    String manufacturerName;
    String categoryName;
    List<String> imageUrl;
}
