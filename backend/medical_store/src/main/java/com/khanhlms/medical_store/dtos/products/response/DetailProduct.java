package com.khanhlms.medical_store.dtos.products.response;

import com.fasterxml.jackson.annotation.JsonFormat;
import com.khanhlms.medical_store.dtos.manufacturer.response.ManufacturerResponse;
import com.khanhlms.medical_store.dtos.questions.response.QuestionResponse;
import lombok.*;
import lombok.experimental.FieldDefaults;

import java.util.Date;
import java.util.List;

@Getter
@Setter
@NoArgsConstructor
@AllArgsConstructor
@FieldDefaults(level = AccessLevel.PRIVATE)
@Builder
public class DetailProduct {

    String id;
    String name;
    String description;
    String category;
    List<String> images;
    String manufacturer;
    String manufactureId;
    Boolean precription;
    Double originPrice;
//    Double discount;
    Double ratingAvg;
    @JsonFormat( pattern = "dd/MM/yyyy")
    Date productDate;
    @JsonFormat(pattern = "dd/MM/yyyy")
    Integer quantity;
    String currency;
    Integer soldQuantity;
    String usage; // cách sử dụng
    List<IngredientResponse> ingredients;
    String benefit; // lợi ích
    String sideEffect; // tác dụng phụ
    String note;
    String preserve;
    List<QuestionResponse>  questions;



}
