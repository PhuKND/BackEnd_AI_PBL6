package com.khanhlms.medical_store.dtos.products.response;

import com.fasterxml.jackson.annotation.JsonFormat;
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
public class ProductResponse {
    String id;
    String name;
    String description;
    Double originPrice;
    Integer quantity;
    Integer soldQuantity;
    Double discountPercent;
    Double discountPrice;
    Integer ratingAvg;
    String imageUrl;
    @JsonFormat( pattern = "dd/MM/yyyy")
    Date productDate;
    @JsonFormat(pattern = "dd/MM/yyyy")
    Date expirationDate;

}
