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
@Table(name = "Ingredients")
@Builder
public class IngredientEntity {
    @Id
    @GeneratedValue(strategy = GenerationType.UUID)
    String id;
    String name;
    Double amount;
    String unit;
    @Column(columnDefinition = "TEXT")
    String description;

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "product_id", nullable = true)
    ProductsEntity product;
}
