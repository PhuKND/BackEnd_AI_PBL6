package com.khanhlms.medical_store.dtos.discounts.request;

import com.fasterxml.jackson.annotation.JsonFormat;
import lombok.*;
import lombok.experimental.FieldDefaults;

import java.time.LocalDateTime;

@AllArgsConstructor
@NoArgsConstructor
@Getter
@Setter
@FieldDefaults(level = AccessLevel.PRIVATE)
public class CreateDiscountRequest {
    Double percent;
    String message;
    @JsonFormat(pattern = "yyyy/MM/dd HH:mm:ss")
    LocalDateTime startDate;
    @JsonFormat(pattern = "yyyy/MM/dd HH:mm:ss")
    LocalDateTime endDate;

}
