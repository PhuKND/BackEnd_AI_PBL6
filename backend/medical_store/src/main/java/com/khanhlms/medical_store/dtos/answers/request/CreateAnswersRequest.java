package com.khanhlms.medical_store.dtos.answers.request;

import lombok.*;
import lombok.experimental.FieldDefaults;

@NoArgsConstructor
@AllArgsConstructor
@Getter
@Setter
@FieldDefaults(level = AccessLevel.PRIVATE)
public class CreateAnswersRequest {
    String answer;
}
