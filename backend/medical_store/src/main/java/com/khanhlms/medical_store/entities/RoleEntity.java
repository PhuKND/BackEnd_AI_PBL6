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
@Table(name = "Roles")
@Builder
public class RoleEntity extends BaseEntity {
    @Id
    @GeneratedValue(strategy = GenerationType.UUID)
    String id;
    String name;

    @ManyToMany(cascade = CascadeType.ALL, fetch = FetchType.EAGER, mappedBy = "roles")
    Set<UserEntity> users;
}
