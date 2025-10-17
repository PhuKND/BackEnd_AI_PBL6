package com.khanhlms.medical_store.entities;
import jakarta.persistence.*;
import lombok.*;
import lombok.experimental.FieldDefaults;

import java.time.LocalDate;
import java.time.LocalDateTime;

@Entity
@Table(name = "discounts")
@Getter
@Setter
@NoArgsConstructor
@AllArgsConstructor
@Builder
@FieldDefaults(level = AccessLevel.PRIVATE)
public class DiscountEntity {

    @Id
    @GeneratedValue(strategy = GenerationType.UUID)
    String id;

    @Column(nullable = false)
    Double percent;
    String message;

    LocalDateTime startDate;
    LocalDateTime endDate;

    @OneToOne
    @JoinColumn(name = "product_id", nullable = false, unique = true)
    ProductsEntity product;
}
