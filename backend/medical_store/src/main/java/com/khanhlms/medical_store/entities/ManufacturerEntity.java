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
@Table(name = "manufacturers")
@Builder
public class ManufacturerEntity extends BaseEntity {
    @Id
    @GeneratedValue(strategy = GenerationType.UUID)
    String id;

    @Column(nullable = false, unique = true)
    String name;

    String country;
    String address;
    String city;
    String phone;
    String email;
    Date foundingDate;
    String thumbnailUrl;
    Boolean active;
    Boolean deleted;

    @Column(columnDefinition = "TEXT")
    String description;

    @OneToMany(mappedBy = "manufacturer", cascade = CascadeType.ALL, orphanRemoval = true)
    Set<ProductsEntity> products;
}
