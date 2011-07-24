package ToyBox::PassiveAggressive;

use strict;
use warnings;

use Data::Dumper;

our $VERSION = '0.0.1';

sub new {
    my $class = shift;
    my $self = {
        dnum => 0,
        data => [],
        lindex => {},
        rlindex => {},
        weight => {},
    } ;
    bless $self, $class;
}

sub add_instance {
    my ($self, %params) = @_;

    my $attributes = $params{attributes} or die "No params: attributes";
    my $label      = $params{label}     or die "No params: label";
    die "attributes is not hash ref"   unless ref($attributes) eq 'HASH';
    die "attributes is empty hash ref" unless keys %$attributes;
    $label = [$label] unless ref($label) eq 'ARRAY';

    my %copy_attr = %$attributes;
    my $squared_norm = 0;
    $squared_norm += $_ ** 2 for values %copy_attr;

    foreach my $l (@$label) {
        if (! defined($self->{lindex}{$l})) {
            my $index = scalar(keys %{$self->{lindex}});
            $self->{lindex}{$l} = $index;
            $self->{rlindex}{$index} = $l;
        }
        my $datum = {f => \%copy_attr, l => $self->{lindex}{$l}, sn => $squared_norm};
        push(@{$self->{data}}, $datum);
        $self->{dnum}++;
    }
}

sub train {
    my ($self, %params) = @_;

    my $T = $params{T} || 10;
    die "T is le 0" unless scalar($T) > 0;

    my $progress_cb = $params{progress_cb};
    my $verbose = 0;
    $verbose = 1 if defined($progress_cb) && $progress_cb eq 'verbose';

    my $algorithm = $params{algorithm} || "PA";
    my $func_calc_tau = $algorithm eq "PA1" ? \&calc_tau_pa1 :
                        $algorithm eq "PA2" ? \&calc_tau_pa2 :
                        \&calc_tau_pa;
    my $C = $params{C} || 1;
    die "C is le 0" unless scalar($C) > 0;

    my $weight = $self->{weight};

    foreach my $t (1 .. $T) {
        my $miss_num = 0;
        foreach my $datum (@{$self->{data}}) {
            my $attributes = $datum->{f};
            my $label = $datum->{l};
            my $squared_norm = $datum->{sn};

            my $scores = {};
            while (my ($f, $val) = each %$attributes) {
                next unless $weight->{$f};
                while (my ($l, $w) = each %{$weight->{$f}}) {
                    $scores->{$l} += $val * $w;
                }
            }

            foreach my $l (keys %{$self->{rlindex}}) {
                my $x = ($label eq $l) ? 1 : -1;
                my $score = $scores->{$l} || 0;
                my $loss = 1 - $x * $score;
                next unless $loss > 0;

                my $tau = $func_calc_tau->($loss, $squared_norm, $C);
                while (my ($f, $v) = each %$attributes) {
                    $weight->{$f}{$l} += $tau * $x * $v;
                }
                $miss_num++;
            }
        }
        print STDERR "t: $t, miss num: $miss_num\n" if $verbose;
        last if $miss_num == 0;
    }
    $self->{weight} = $weight;

    1;
}

sub predict {
    my ($self, %params) = @_;

    my $attributes = $params{attributes} or die "No params: attributes";
    die "attributes is not hash ref"   unless ref($attributes) eq 'HASH';
    die "attributes is empty hash ref" unless keys %$attributes;

    my $weight = $self->{weight};
    my $rlindex = $self->{rlindex};

    my $scores = {};
    while (my ($f, $val) = each %$attributes) {
        while (my ($l, $w) = each %{$weight->{$f}}) {
            $scores->{$rlindex->{$l}} += $val * $w;
        }
    }

    $scores;
}

sub labels {
    my $self = shift;
    keys %{$self->{lindex}};
}

sub calc_tau_pa {
    my ($l, $squared_norm) = @_;
    $l / $squared_norm;
}

sub calc_tau_pa1 {
    my ($l, $squared_norm, $C) = @_;
    my $tmp = $l / $squared_norm;
    $tmp < $C ? $tmp : $C;
}

sub calc_tau_pa2 {
    my ($l, $squared_norm, $C) = @_;
    $l / ($squared_norm + (1 / (2 * $C)));
}

1;
__END__


=head1 NAME

ToyBox::PassiveAggressive - Classifier using PassiveAggressive Algorithm

=head1 SYNOPSIS

  use ToyBox::PassiveAggressive;

  my $pa= ToyBox::PassiveAggressive->new();
  
  $pa->add_instance(
      attributes => {a => 2, b => 3},
      label => 'positive'
  );
  
  $pa->add_instance(
      attributes => {c => 3, d => 1},
      label => 'negative'
  );
  
  $pa->train(T => 10,
             algorithm => 'PA2',
             C => 1.0,
             progress_cb => 'verbose');
  
  my $score = $pa->predict(
                  attributes => {a => 1, b => 1, d => 1, e =>1}
              );

=head1 DESCRIPTION

=head1 AUTHOR

TAGAMI Yukihiro <tagami.yukihiro@gmail.com>

=head1 LICENSE

This library is distributed under the term of the MIT license.

L<http://opensource.org/licenses/mit-license.php>

