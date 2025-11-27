package com.khanhlms.medical_store.entities;

import jakarta.persistence.*;
import lombok.*;
import lombok.experimental.FieldDefaults;

@Entity
@Getter
@Setter
@ToString
@NoArgsConstructor
@AllArgsConstructor
@Builder
@FieldDefaults(level = AccessLevel.PRIVATE)
@Table(name = "Order_Items")
public class OrderItemEntity extends BaseEntity {
    @Id
    @GeneratedValue(strategy = GenerationType.UUID)
    String id;
    @ManyToOne
    @JoinColumn(name = "product_id")
    ProductsEntity product;
    // price when user buy product
    Double price;
    Double discountPrice;
    // Số lượng mua
    Integer quantity;
    Double total;

    @ManyToOne
    @JoinColumn(name = "order_id")
    OrderEntity order;
}
