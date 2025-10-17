package com.khanhlms.medical_store.entities;

import jakarta.persistence.*;
import lombok.*;
import lombok.experimental.FieldDefaults;

import java.time.LocalDate;
import java.util.Date;
import java.util.Set;

@Setter
@Getter
@FieldDefaults(level = AccessLevel.PRIVATE)
@NoArgsConstructor
@AllArgsConstructor
@Entity
@Table(name = "frequentlys_question_products")
@Builder

public class FrequentlyEntity {
    @Id
    @GeneratedValue(strategy = GenerationType.UUID)
    String id;
    @Column(columnDefinition = "TEXT")
    String question;
    @Column(columnDefinition = "TEXT")
    String answer;

    LocalDate createdDate;
    @ManyToOne
    @JoinColumn(name = "product_id", nullable = false)
    ProductsEntity  product;
}
