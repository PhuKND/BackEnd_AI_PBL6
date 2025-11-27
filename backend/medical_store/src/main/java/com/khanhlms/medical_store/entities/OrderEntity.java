package com.khanhlms.medical_store.entities;

import jakarta.persistence.*;
import lombok.*;
import lombok.experimental.FieldDefaults;

import java.util.List;

@Entity
@Getter
@Setter
@Builder
@NoArgsConstructor
@AllArgsConstructor
@Table(name = "Orders")
@FieldDefaults(level = AccessLevel.PRIVATE)
public class OrderEntity extends BaseEntity {
    @Id
    @GeneratedValue(strategy = GenerationType.UUID)
    String id;
    // User mua hàng
    @ManyToOne
    @JoinColumn(name = "user_id")
    UserEntity user;

    @OneToMany(mappedBy = "order", cascade = CascadeType.ALL)
    List<OrderItemEntity> orderItems;
    Double totalAmount;
    String status;
    String address;
    String city;
    String ward;
    @Column(columnDefinition = "TEXT")
    String note;
    String phoneNumber;
    @OneToOne(mappedBy = "order", cascade = CascadeType.ALL)
    PaymentEntity payment;
}
