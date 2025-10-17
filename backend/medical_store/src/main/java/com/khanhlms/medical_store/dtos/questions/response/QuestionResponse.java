package com.khanhlms.medical_store.dtos.questions.response;

import com.fasterxml.jackson.annotation.JsonFormat;
import com.khanhlms.medical_store.dtos.answers.response.AnswerResponse;
import jakarta.persistence.FetchType;
import jakarta.persistence.ManyToOne;
import lombok.*;
import lombok.experimental.FieldDefaults;

import java.util.Date;
import java.util.List;

@NoArgsConstructor
@AllArgsConstructor
@Getter
@Setter
@Builder
@FieldDefaults(level = AccessLevel.PRIVATE)
public class QuestionResponse {
    String questionId;
    String question;
    @JsonFormat(shape = JsonFormat.Shape.STRING,pattern = "yyyy-MM-dd HH:mm:ss")
    Date createdDate;
    Integer likesCount;
    String userName;
    String avatarUrl;
    List<AnswerResponse> answers;

}
