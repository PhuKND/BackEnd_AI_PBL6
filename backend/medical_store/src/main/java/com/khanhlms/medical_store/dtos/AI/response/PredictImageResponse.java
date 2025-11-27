package com.khanhlms.medical_store.dtos.AI.response;

import lombok.*;
import lombok.experimental.FieldDefaults;

@NoArgsConstructor
@AllArgsConstructor
@Getter
@Setter
@FieldDefaults(level = AccessLevel.PRIVATE)
@Builder
@ToString
public class PredictImageResponse {
    Integer class_index;
    String name_en;
    String name_vi;
    Double confidence;
}
