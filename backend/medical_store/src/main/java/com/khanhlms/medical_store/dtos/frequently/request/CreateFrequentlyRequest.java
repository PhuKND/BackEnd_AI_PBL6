package com.khanhlms.medical_store.dtos.frequently.request;


import lombok.*;
import lombok.experimental.FieldDefaults;

@NoArgsConstructor
@AllArgsConstructor
@Getter
@Setter
@FieldDefaults(level = AccessLevel.PRIVATE)
public class CreateFrequentlyRequest {
    String question;
    String answer;
}
