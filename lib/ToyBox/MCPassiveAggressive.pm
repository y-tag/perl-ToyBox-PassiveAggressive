package ToyBox::MCPassiveAggressive;

use strict;
use warnings;

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

    my %copy_attr  = %$attributes;

    my @num_label = ();
    foreach my $l (@$label) {
        if (! defined($self->{lindex}{$l})) {
            my $index = scalar(keys %{$self->{lindex}});
            $self->{lindex}{$l} = $index;
            $self->{rlindex}{$index} = $l;
        }
        push @num_label, $self->{lindex}{$l};
    }

    my $squared_norm = 0;
    $squared_norm += $_ ** 2 for values %copy_attr;

    my $datum = {f => \%copy_attr, l => \@num_label, sn => $squared_norm};
    push @{$self->{data}}, $datum;
    $self->{dnum}++;
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

    my @all_labels = values %{$self->{lindex}};
    my $weight = $self->{weight};

    foreach my $t (1 .. $T) {
        my $miss_num = 0;
        foreach my $datum (@{$self->{data}}) {
            my $attributes = $datum->{f};
            my $squared_norm = $datum->{sn};

            my %labels_hash = map {$_ => 1} @{$datum->{l}};
            my @unlabeled = grep {! $labels_hash{$_}} @all_labels;

            die "The datum which has all kind of labels exists\n"
                unless @unlabeled;

            my $lscores = {}; # labeled score
            my $uscores = {}; # unlabeled score
            while (my ($f, $val) = each %$attributes) {
                next unless $weight->{$f};
                while (my ($l, $w) = each %{$weight->{$f}}) {
                    if (defined($labels_hash{$l})) {
                        $lscores->{$l} += $val * $w;
                    } else {
                        $uscores->{$l} += $val * $w;
                    }
                }
            }

            my @asc_labeled = 
                sort {$lscores->{$a} <=> $lscores->{$b}} keys %$lscores;
            my @desc_unlabeled = 
                sort {$uscores->{$b} <=> $uscores->{$a}} keys %$uscores;
            my $minl = @asc_labeled    ? $asc_labeled[0]    : $datum->{l}[0]; 
            my $maxu = @desc_unlabeled ? $desc_unlabeled[0] : $unlabeled[0];
            my $minl_score = $lscores->{$minl} || 0;
            my $maxu_score = $uscores->{$maxu} || 0;

            my $l = 1 - $minl_score + $maxu_score;
            next unless $l > 0;

            my $tau = $func_calc_tau->($l, $squared_norm, $C);
            while (my ($f, $val) = each %$attributes) {
                $weight->{$f}{$minl} += $tau * $val;
                $weight->{$f}{$maxu} -= $tau * $val;
            }
            $miss_num += 1;
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
    $l / (2 * $squared_norm);
}

sub calc_tau_pa1 {
    my ($l, $squared_norm, $C) = @_;
    my $tmp = $l / (2 * $squared_norm);
    $tmp < $C ? $tmp : $C;
}

sub calc_tau_pa2 {
    my ($l, $squared_norm, $C) = @_;
    $l / (2 * $squared_norm + (1 / (2 * $C)));
}


1;
__END__


=head1 NAME

ToyBox::MCPassiveAggressive - Classifier using MultiClass PassiveAggressive Algorithm

=head1 SYNOPSIS

  use ToyBox::MCPassiveAggressive;

  my $pa= ToyBox::MCPassiveAggressive->new();
  
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

