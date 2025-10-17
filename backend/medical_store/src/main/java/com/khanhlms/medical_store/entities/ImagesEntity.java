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
@Table(name = "Images")
@Builder
public class ImagesEntity {
    @Id
    @GeneratedValue(strategy = GenerationType.UUID)
    String id;
    String imageUrl;

    @ManyToOne
    @JoinColumn(name = "product_id", nullable = false)
    ProductsEntity  product;
}
