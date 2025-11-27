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
@Table(name = "Products")
@Builder
public class ProductsEntity extends BaseEntity {
    @Id
    @GeneratedValue(strategy = GenerationType.UUID)
    String id;
    String name;
    @Column(columnDefinition = "TEXT")
    String description;
    Double originPrice;
    @OneToOne(mappedBy = "product", cascade = CascadeType.ALL, fetch = FetchType.LAZY)
    DiscountEntity discount;
    Boolean precription;
    Double ratingAvg;
    String unit;
    @Column(nullable = false)
    Date productDate;
    @Column(nullable = false)
    Date expirationDate;
    Integer quantity;
    String currency;
    Integer soldQuantity;
    Integer position;
    String sideEffect; // tác dụng phụ
    String note;
    String preserve; // cach bao quan
    @Column(name = "product_usage")
    String usage; // cách sử dụng
    String benefit; // lợi ích
    @OneToMany(mappedBy = "product", cascade = CascadeType.ALL, orphanRemoval = true, fetch = FetchType.EAGER)
    List<IngredientEntity> ingredients;
//
    Boolean isActive;
    Boolean isDeleted;
//

    @ManyToOne(fetch = FetchType.EAGER)
    @JoinColumn(name = "manufacturer_id", nullable = false)
    ManufacturerEntity manufacturer;

    @ManyToOne(fetch = FetchType.EAGER)
    @JoinColumn(name = "category_id", nullable = false)
    CategoryEntity category;

    @OneToMany(mappedBy = "product", cascade = CascadeType.ALL, orphanRemoval = true, fetch = FetchType.EAGER)
    List<ImagesEntity> images;

    @OneToMany(mappedBy = "product", cascade = CascadeType.ALL, orphanRemoval = true, fetch = FetchType.EAGER)
    List<QuestionEntity> questions;

    @OneToMany(mappedBy = "product", cascade = CascadeType.ALL, orphanRemoval = true, fetch = FetchType.EAGER)
    List<FrequentlyEntity>  frequently;

    @OneToMany(mappedBy = "product", cascade = CascadeType.ALL, orphanRemoval = true)
    private List<CartItemEntity> cartItems;

}
