package com.khanhlms.medical_store.dtos.answers.response;

import com.fasterxml.jackson.annotation.JsonFormat;
import lombok.*;
import lombok.experimental.FieldDefaults;

import java.util.Date;

@NoArgsConstructor
@AllArgsConstructor
@Getter
@Setter
@FieldDefaults(level = AccessLevel.PRIVATE)
@Builder
public class AnswerResponse {
    String answerId;
    String answer;
    @JsonFormat(shape = JsonFormat.Shape.STRING,pattern = "yyyy-MM-dd HH:mm:ss")
    Date createdDate;
    Integer likesCount;
    String userName;
    String avatarUrl;
}
