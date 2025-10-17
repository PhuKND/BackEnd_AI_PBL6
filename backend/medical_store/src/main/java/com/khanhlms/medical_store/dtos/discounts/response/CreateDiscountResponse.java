package com.khanhlms.medical_store.dtos.discounts.response;

import java.time.LocalDateTime;

import com.fasterxml.jackson.annotation.JsonFormat;
import lombok.*;
import lombok.experimental.FieldDefaults;

import java.time.LocalDateTime;

@AllArgsConstructor
@NoArgsConstructor
@Getter
@Setter
@FieldDefaults(level = AccessLevel.PRIVATE)
@Builder
public class CreateDiscountResponse {
    String productId;
    Double percent;
    String message;
    @JsonFormat(pattern = "yyyy/MM/dd HH:mm:ss")
    LocalDateTime startDate;
    @JsonFormat(pattern = "yyyy/MM/dd HH:mm:ss")
    LocalDateTime endDate;

}
