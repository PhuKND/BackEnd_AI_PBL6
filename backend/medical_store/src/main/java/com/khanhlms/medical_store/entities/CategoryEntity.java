package com.khanhlms.medical_store.entities;

import jakarta.persistence.*;
import lombok.*;
import lombok.experimental.FieldDefaults;

import java.util.Set;

@Setter
@Getter
@FieldDefaults(level = AccessLevel.PRIVATE)
@NoArgsConstructor
@AllArgsConstructor
@Entity
@Table(name = "Categories")
@Builder
public class CategoryEntity extends BaseEntity {
    @Id
    @GeneratedValue(strategy = GenerationType.UUID)
    String id;
    @Column(nullable = false)
    String name;

    @Column(columnDefinition = "TEXT")
    String description;
    @Column(columnDefinition = "TEXT")
    String thumbnailUrl;
    Integer position;
    Boolean active;
    Boolean deleted;
    @Column(columnDefinition = "TEXT")
    @OneToMany(mappedBy = "category", cascade = CascadeType.ALL,  orphanRemoval = true)
    Set<ProductsEntity> products;

}
