package com.khanhlms.medical_store.entities;


import jakarta.persistence.*;
import lombok.*;
import lombok.experimental.FieldDefaults;

import java.util.Date;
import java.util.List;

@Setter
@Getter
@FieldDefaults(level = AccessLevel.PRIVATE)
@NoArgsConstructor
@AllArgsConstructor
@Entity
@Table(name = "questions")
@Builder
public class QuestionEntity {
    @Id
    @GeneratedValue(strategy = GenerationType.UUID)
    String id;
    @Column(columnDefinition = "TEXT", nullable = false)
    String question;
    Date createdDate;
    Integer likesCount;
    @ManyToOne(fetch = FetchType.LAZY)
            @JoinColumn(name = "user_id", nullable = false)
    UserEntity user;
    @ManyToOne(fetch = FetchType.LAZY)
            @JoinColumn(name = "product_id", nullable = false)
    ProductsEntity product;

    @OneToMany(mappedBy = "question", cascade = CascadeType.ALL, orphanRemoval = true, fetch = FetchType.EAGER)
    List<AnswersEntity> answers;
}
