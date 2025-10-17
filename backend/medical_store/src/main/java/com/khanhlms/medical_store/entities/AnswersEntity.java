package com.khanhlms.medical_store.entities;

import jakarta.persistence.*;
import lombok.*;
import lombok.experimental.FieldDefaults;

import java.util.Date;
import java.util.Set;

@Setter
@Getter
@FieldDefaults(level = AccessLevel.PRIVATE)
@NoArgsConstructor
@AllArgsConstructor
@Entity
@Table(name = "Answers")
@Builder
public class AnswersEntity {
    @Id
    @GeneratedValue(strategy = GenerationType.UUID)
    String id;
    @Column(nullable = false, columnDefinition = "TEXT")
    String answer;
    Date createdDate;
    Integer likesCount;

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "user_id",  nullable = false)
    UserEntity user;

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "question_id", nullable = false)
    QuestionEntity question;
}
